pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../layout" as Room36Layout
import "../theme" as Room36Theme
import "../dev" as Dev
import "../components" as Components
import "../navigation/LibraryRoutes.js" as Routes

Rectangle {
    id: root
    objectName: "libraryShell"

    property var projectId: null
    property var photoModel: null
    property var processingController: null
    property var peopleController: null
    property var duplicateController: null
    property var favoritesController: null
    property var trashController: null
    property var exportController: null
    property var inspectorController: null
    property string libraryViewSize: "Medium"
    property string blurViewSize: "Medium"
    property string knownPeopleViewSize: "Medium"
    property string unknownPeopleViewSize: "Medium"
    property string duplicateViewSize: "Medium"
    property string favoritesViewSize: "Medium"
    property string trashViewSize: "Medium"
    property string projectName: ""
    property string currentRoute: "library"
    property var routeCounts: ({
        blurry: processingController ? processingController.routeCounts.blurry : -1,
        duplicates: duplicateController ? duplicateController.needsReviewCount : -1,
        favorites: favoritesController ? favoritesController.count : -1,
        trash: trashController ? trashController.count : -1,
        knownPeople: peopleController ? peopleController.knownCount : -1,
        unknownPeople: peopleController ? peopleController.unknownCount : -1
    })
    readonly property var currentEntry: Routes.entry(currentRoute)
    readonly property var routeStage: processingController && processingController.stages.length > 0
        ? processingController.stages[currentRoute === "duplicates" ? 1 : 5] : ({state: "NOT_STARTED"})

    signal homeRequested()
    signal settingsRequested()
    signal routeSelected(string routeId)
    signal sortRequested(string routeId, string option)
    signal viewSizeRequested(string routeId, string option)

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    color: theme.libraryBackground
    implicitWidth: metrics.minimumWindowWidth
    implicitHeight: metrics.minimumWindowHeight

    function showRoute(routeId) {
        if (inspectorController) inspectorController.close()
        currentRoute = Routes.normalize(routeId)
    }

    onCurrentRouteChanged: {
        const normalized = Routes.normalize(currentRoute)
        if (currentRoute !== normalized) {
            currentRoute = normalized
            return
        }
        routeSelected(currentRoute)
    }

    Room36Layout.LibrarySidebar {
        id: sidebar
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: root.width >= metrics.libraryWideBreakpoint
            ? metrics.sidebarPreferredWidth : metrics.sidebarCompactWidth
        currentRoute: root.currentRoute
        routeCounts: root.routeCounts
        onRouteRequested: function(routeId) { root.showRoute(routeId) }
        onHomeRequested: root.homeRequested()
        onSettingsRequested: root.settingsRequested()
    }

    Room36Layout.LibraryTopBar {
        id: topBar
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: parent.top
        height: implicitHeight
        projectName: root.projectName
        pageTitle: root.currentEntry.title
        libraryControlsEnabled: ["library", "blurry", "duplicates", "favorites", "knownPeople", "unknownPeople", "trash"].indexOf(root.currentRoute) >= 0
        sortOptions: root.currentRoute === "blurry"
            ? ["Most problematic first", "Least problematic first", "Filename A–Z", "Filename Z–A"]
            : root.currentRoute === "duplicates"
                ? ["Largest groups", "Smallest groups"]
            : root.currentRoute === "knownPeople"
                ? ["Name A–Z", "Most photos", "Most faces"]
                : root.currentRoute === "unknownPeople"
                    ? ["Largest clusters", "Smallest clusters"]
                    : ["Name A–Z", "Name Z–A"]
        onHomeRequested: root.homeRequested()
        onSortRequested: function(option) {
            if (root.currentRoute === "library" && root.photoModel && ["Name A–Z", "Name Z–A"].indexOf(option) >= 0)
                root.photoModel.sortByFilename(option === "Name Z–A")
            else if (root.currentRoute === "favorites" && root.favoritesController)
                root.favoritesController.sortByFilename(option === "Name Z–A")
            else if (root.currentRoute === "trash" && root.trashController)
                root.trashController.photoModel.sortByFilename(option === "Name Z–A")
            else if (root.currentRoute === "blurry" && root.processingController)
                root.processingController.blurModel.setSort(option)
            else if (root.currentRoute === "duplicates" && root.duplicateController)
                root.duplicateController.groupModel.setSort(option)
            else if (root.currentRoute === "knownPeople" && root.peopleController)
                root.peopleController.knownModel.setSort(option)
            else if (root.currentRoute === "unknownPeople" && root.peopleController)
                root.peopleController.unknownModel.setSort(option)
            root.sortRequested(root.currentRoute, option)
        }
        onViewSizeRequested: function(option) {
            if (["Small", "Medium", "Large"].indexOf(option) >= 0) {
                if (root.currentRoute === "library") root.libraryViewSize = option
                else if (root.currentRoute === "favorites") root.favoritesViewSize = option
                else if (root.currentRoute === "trash") root.trashViewSize = option
                else if (root.currentRoute === "blurry") root.blurViewSize = option
                else if (root.currentRoute === "duplicates") root.duplicateViewSize = option
                else if (root.currentRoute === "knownPeople") root.knownPeopleViewSize = option
                else if (root.currentRoute === "unknownPeople") root.unknownPeopleViewSize = option
            }
            root.viewSizeRequested(root.currentRoute, option)
        }
    }

    Loader {
        id: contentLoader
        objectName: "libraryContentLoader"
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: processingStatus.bottom
        anchors.bottom: parent.bottom
        anchors.margins: metrics.libraryContentMargin

        sourceComponent: root.currentRoute === "library" ? gridComponent
            : root.currentRoute === "blurry" && root.processingController ? blurComponent
            : root.currentRoute === "duplicates" ? duplicateComponent
            : root.currentRoute === "favorites" ? favoritesComponent
            : root.currentRoute === "trash" ? trashComponent
            : root.currentRoute === "knownPeople" ? knownComponent
            : root.currentRoute === "unknownPeople" ? unknownComponent : placeholderComponent
    }

    Components.ProcessingStatus {
        id: processingStatus
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: topBar.bottom
        controller: root.processingController
    }

    Connections {
        target: root.exportController
        function onPlanReady() { exportDialog.open() }
    }
    Components.ExportDialog {
        id: exportDialog
        exporter: root.exportController
    }
    Text {
        objectName: "exportShellError"
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        z: 11
        visible: root.exportController !== null && root.exportController.error.length > 0
            && !exportDialog.visible
        text: root.exportController ? root.exportController.error : ""
        color: theme.mainBlue
        Accessible.name: text
    }

    Component {
        id: favoritesComponent
        FavoritesPage {
            favorites: root.favoritesController
            exporter: root.exportController
            inspector: root.inspectorController
            viewSize: root.favoritesViewSize
        }
    }

    Component {
        id: trashComponent
        TrashPage {
            trash: root.trashController
            favorites: root.favoritesController
            inspector: root.inspectorController
            viewSize: root.trashViewSize
        }
    }

    Component {
        id: blurComponent
        BlurryPhotosPage {
            processing: root.processingController
            exporter: root.exportController
            inspector: root.inspectorController
            favorites: root.favoritesController
            viewSize: root.blurViewSize
        }
    }

    Component {
        id: duplicateComponent
        NearDuplicatesPage {
            duplicates: root.duplicateController
            exporter: root.exportController
            inspector: root.inspectorController
            favorites: root.favoritesController
            viewSize: root.duplicateViewSize
        }
    }

    Component {
        id: knownComponent
        KnownPeoplePage {
            people: root.peopleController
            exporter: root.exportController
            inspector: root.inspectorController
            favorites: root.favoritesController
            viewSize: root.knownPeopleViewSize
        }
    }

    Component {
        id: unknownComponent
        UnknownPeoplePage {
            people: root.peopleController
            exporter: root.exportController
            inspector: root.inspectorController
            favorites: root.favoritesController
            viewSize: root.unknownPeopleViewSize
        }
    }

    Component {
        id: gridComponent
        ColumnLayout {
            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }
                Components.PillButton {
                    objectName: "libraryExportSelected"
                    variant: "accent"
                    text: "Export selected"
                    enabled: root.exportController !== null && !root.exportController.busy
                        && !root.exportController.preparing && root.photoModel !== null
                        && root.photoModel.selectedCount > 0
                    onClicked: root.exportController.prepareSelected(root.photoModel)
                    Accessible.name: "Export selected library photos"
                }
            }
            Components.PhotoGrid {
                Layout.fillWidth: true
                Layout.fillHeight: true
                photoModel: root.photoModel
                favoritesController: root.favoritesController
                viewSize: root.libraryViewSize
            }
        }
    }

    Text {
        objectName: "favoritesErrorMessage"
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        z: 10
        visible: root.currentRoute !== "favorites" && root.favoritesController !== null
            && root.favoritesController.error.length > 0
        text: root.favoritesController ? root.favoritesController.error : ""
        color: theme.mainBlue
        Accessible.name: text
    }

    Component {
        id: placeholderComponent
        Dev.LibraryRoutePlaceholder {
            routeId: root.currentRoute
            title: root.currentEntry.title
            message: root.currentEntry.message + (root.processingController && ["duplicates", "knownPeople", "unknownPeople"].indexOf(root.currentRoute) >= 0
                ? "\n" + (root.routeStage.state === "COMPLETE"
                    ? "Analysis is available. Review tools will arrive in a later phase."
                    : "Analysis status: " + root.routeStage.state)
                : "")
        }
    }
}
