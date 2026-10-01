import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme
import "../components" as Components
import "../layout" as Room36Layout

Item {
    id: root
    objectName: "homePage"

    property var projectModel: null
    property bool newProjectEnabled: true
    property string searchText: ""

    signal newProjectRequested()
    signal projectActivated(var projectId, string projectName)
    signal projectMenuRequested(var projectId)
    signal searchChanged(string text)
    signal sortChanged(string value)
    signal viewSizeChanged(string value)
    signal subscriptionRequested()
    signal settingsRequested()
    signal profileRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    readonly property bool referenceLayout: width >= 1500
    readonly property real pageMargin: referenceLayout ? metrics.homeHorizontalMargin : 48
    readonly property real headingTop: referenceLayout ? metrics.homeHeadingTop : 174
    readonly property real gridTop: referenceLayout ? metrics.homeGridTop : 238

    Rectangle {
        anchors.fill: parent
        color: theme.homeBackground
    }

    Room36Layout.HomeTopBar {
        id: topBar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        height: implicitHeight
        searchText: root.searchText
        onSearchChanged: function(text) { root.searchChanged(text) }
        onSubscriptionRequested: root.subscriptionRequested()
        onSettingsRequested: root.settingsRequested()
        onProfileRequested: root.profileRequested()
    }

    RowLayout {
        id: homeLabel
        x: root.pageMargin
        y: root.referenceLayout ? 246 : root.headingTop
        width: 260
        height: metrics.toolbarButtonHeight
        spacing: metrics.spacingSmall

        Image {
            Layout.preferredWidth: metrics.iconSizeLarge
            Layout.preferredHeight: metrics.iconSizeLarge
            source: theme.iconUrl("home.svg")
            cache: true
            fillMode: Image.PreserveAspectFit
        }

        Text {
            text: "Home"
            color: theme.mainBlue
            font.family: typography.pageTitleFamily
            font.pixelSize: typography.sectionTitle
            verticalAlignment: Text.AlignVCenter
        }
    }

    RowLayout {
        id: headingActions
        x: root.referenceLayout
            ? root.width - implicitWidth - 84
            : Math.max(root.pageMargin, root.width - implicitWidth - root.pageMargin)
        y: root.referenceLayout ? 241.63 : root.headingTop
        spacing: metrics.spacingMedium

        Components.ToolbarDropdownButton {
            objectName: "homeSortButton"
            label: "Sort by"
            iconSource: theme.iconUrl("sort.svg")
            options: ["Newest", "Oldest", "Name"]
            onOptionSelected: function(option) { root.sortChanged(option) }
        }

        Components.ToolbarDropdownButton {
            objectName: "homeViewButton"
            enabled: false // Project-card size variants are not implemented.
            label: "View"
            iconSource: theme.iconUrl("view.svg")
            options: ["Small", "Medium", "Large"]
            onOptionSelected: function(option) { root.viewSizeChanged(option) }
        }
    }

    Components.ProjectGrid {
        id: projectGrid
        x: root.pageMargin
        y: root.gridTop
        width: root.width - root.pageMargin * 2
        height: Math.max(0, root.height - y - 40)
        projectModel: root.projectModel
        newProjectEnabled: root.newProjectEnabled
        onNewProjectRequested: root.newProjectRequested()
        onProjectActivated: function(projectId, projectName) { root.projectActivated(projectId, projectName) }
        onProjectMenuRequested: function(projectId) { root.projectMenuRequested(projectId) }
    }
}
