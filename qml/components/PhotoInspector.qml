pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

FocusScope {
    id: root
    objectName: "photoInspector"
    property var controller: null
    property var favoritesController: null
    property Component detailsComponent: null
    property bool showReviewActions: true
    readonly property var photo: controller ? controller.photo : ({})
    property bool currentFavorite: false
    property bool favoritePending: false
    function refreshFavorite() {
        const path = photo.photoId || ""
        currentFavorite = favoritesController && path ? favoritesController.isFavorite(path) : false
        favoritePending = favoritesController && path ? favoritesController.isPending(path) : false
    }
    onPhotoChanged: refreshFavorite()
    onFavoritesControllerChanged: refreshFavorite()
    Connections {
        target: root.favoritesController
        function onFavoriteChanged(path, favorite) {
            if (path === root.photo.photoId) root.currentFavorite = favorite
        }
        function onPendingChanged(path, pending) {
            if (path === root.photo.photoId) root.favoritePending = pending
        }
        function onChanged() { root.refreshFavorite() }
    }
    signal reviewRequested(string identity, string intent)
    Room36Theme.Theme { id: theme }
    onVisibleChanged: if (visible) forceActiveFocus()
    Keys.priority: Keys.BeforeItem
    Keys.onPressed: function(event) {
        if (!root.controller) return
        if (event.key === Qt.Key_Left) { root.controller.previous(); event.accepted = true }
        else if (event.key === Qt.Key_Right) { root.controller.next(); event.accepted = true }
        else if (event.key === Qt.Key_Escape) { root.controller.close(); event.accepted = true }
    }
    Rectangle { anchors.fill: parent; color: theme.panelFill }
    // Consume pointer input so obscured candidate cards cannot receive clicks.
    MouseArea { anchors.fill: parent }
    ColumnLayout {
        anchors.fill: parent
        spacing: 16
        RowLayout {
            Layout.fillWidth: true
            PrimaryButton {
                objectName: "inspectorBack"
                text: "← Back"
                onClicked: if (root.controller) root.controller.close()
                Accessible.name: "Close photo inspector"
            }
            Text {
                Layout.fillWidth: true
                text: root.photo.filename || ""
                color: theme.mainBlue
                elide: Text.ElideMiddle
                font.pixelSize: 22
            }
            IconButton {
                objectName: "inspectorFavoriteButton"
                source: theme.iconUrl(root.currentFavorite ? "favorite_filled.svg" : "favorite.svg")
                accessibleLabel: root.currentFavorite ? "Remove favorite" : "Add favorite"
                visible: root.favoritesController !== null && root.favoritesController.loaded
                    && Boolean(root.photo.photoId)
                enabled: !root.favoritePending
                onClicked: root.favoritesController.setFavorite(root.photo.photoId, !root.currentFavorite)
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 20
            Item {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Rectangle { anchors.fill: parent; color: theme.imagePlaceholder }
                Image {
                    objectName: "inspectorImage"
                    anchors.fill: parent
                    source: root.controller ? root.controller.previewSource : ""
                    cache: false
                    asynchronous: true
                    fillMode: Image.PreserveAspectFit
                    Accessible.name: root.photo.filename || "Photo preview"
                }
                BusyIndicator {
                    anchors.centerIn: parent
                    running: root.controller !== null && root.controller.previewState === "loading"
                    visible: running
                }
                Text {
                    anchors.centerIn: parent
                    visible: root.controller !== null && root.controller.previewState === "error"
                    text: "This photo preview is unavailable."
                    color: theme.mainBlue
                }
            }
            ColumnLayout {
                visible: root.detailsComponent === null
                Layout.preferredWidth: root.width < 900 ? 180 : 240
                Layout.alignment: Qt.AlignTop
                Text { text: "Image quality"; color: theme.mainBlue; font.pixelSize: 22 }
                Text { text: root.photo.classification || ""; color: theme.mainBlue; font.pixelSize: 18 }
                Text {
                    text: "Sharpness quality: " + (root.photo.qualityScore !== undefined ? root.photo.qualityScore.toFixed(2) : "—") + " / 1"
                    color: theme.mutedText
                }
                Text { text: "Higher values mean sharper detail."; color: theme.mutedText; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                Text { text: "Review decision: " + (root.photo.reviewState || "UNDECIDED").toLowerCase(); color: theme.mainBlue }
                Text { text: "Remove marks this photo for a future removal workflow. The original stays in its folder.";
                    color: theme.mutedText; wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
            Loader {
                visible: root.detailsComponent !== null
                Layout.preferredWidth: root.width < 900 ? 180 : 240
                Layout.fillHeight: true
                sourceComponent: root.detailsComponent
            }
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            PrimaryButton {
                objectName: "inspectorPrevious"
                text: "Previous"
                enabled: root.controller !== null && root.controller.hasPrevious
                onClicked: root.controller.previous()
                Accessible.name: "Previous photo"
            }
            Item { Layout.fillWidth: true }
            PrimaryButton {
                objectName: "inspectorKeep"
                text: "Keep"
                visible: root.showReviewActions
                onClicked: root.reviewRequested(root.photo.photoId, "KEEP")
                Accessible.name: "Keep photo"
            }
            PrimaryButton {
                objectName: "inspectorReview"
                text: "Review"
                visible: root.showReviewActions
                onClicked: root.reviewRequested(root.photo.photoId, "REVIEW")
                Accessible.name: "Mark photo for review"
            }
            PrimaryButton {
                objectName: "inspectorRemove"
                text: "Remove"
                visible: root.showReviewActions
                onClicked: root.reviewRequested(root.photo.photoId, "REMOVE")
                Accessible.name: "Mark photo for future removal"
            }
            Item { Layout.fillWidth: true }
            PrimaryButton {
                objectName: "inspectorNext"
                text: "Next"
                enabled: root.controller !== null && root.controller.hasNext
                onClicked: root.controller.next()
                Accessible.name: "Next photo"
            }
        }
    }
}
